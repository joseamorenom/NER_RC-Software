#!/bin/bash
# NER software handler

if [ $# -gt 0 ]
    then
    MODE="$1"
    STANDARD="False"
    FAST="False"
    CUDA="False"
    UFLAG="False"
        if [ ${MODE} == 'TRAIN' ]
        then
            shift # past argument
            if [ $# -gt 1 ]
                then
                while [[ $# -gt 1 ]]; do
                    case $1 in
                        -f|--fast)
                        FAST="$2"
                        shift # past argument
                        shift # past value
                        ;;

                        -m|--model)
                        MODEL="$2"
                        shift # past argument
                        shift # past value
                        ;;

                        -s|--standard)
                        STANDARD="$2"
                        shift # past argument
                        shift # past value
                        ;;

                        -id|--inputdir)
                        INPUTDIR="$2"
                        shift # past argument
                        shift # past value
                        ;;

                        -u|--upsampleflag)
                        UFLAG="$2"
                        shift # past argument
                        shift # past value
                        ;;

                        -cu|--cuda)
                        CUDA="$2"
                        shift # past argument
                        shift # past value
                        ;;

                    esac
                done
                    python src/scripts/Train_model.py -f ${FAST} -m ${MODEL} -s ${STANDARD} -id "${INPUTDIR}" -u "${UFLAG}" -cu "${CUDA}"
            else
                echo Not arguments the script requires at least input directory
            fi


        elif [ $1 == 'USE' ]
        then
        shift # past argument
        if [ $# -gt 1 ]
            then
            while [[ $# -gt 1 ]]; do
                case $1 in
                    -m|--model)
                    MODEL="$2"
                    shift # past argument
                    shift # past value
                    ;;

                    -id|--inputdir)
                    INPUTDIR="$2"
                    shift # past argument
                    shift # past value
                    ;;

                    -od|--outputdir)
                    OUTPUTDIR="$2"
                    shift # past argument
                    shift # past value
                    ;;

                    -cu|--cuda)
                    CUDA="$2"
                    shift # past argument
                    shift # past value
                    ;;

                esac

            done
            if [ -n "${OUTPUTDIR}" ] && [ -n "${CUDA}" ]; then
                python src/scripts/Tagged_document.py -m ${MODEL} -id "${INPUTDIR}" -od "${OUTPUTDIR}" -cu "${CUDA}"

            elif [[ -n "${OUTPUTDIR}" ]]; then
                python src/scripts/Tagged_document.py -m ${MODEL} -id "${INPUTDIR}" -od "${OUTPUTDIR}" 

            elif [[ -n "${CUDA}" ]]; then
                python src/scripts/Tagged_document.py -m ${MODEL} -id "${INPUTDIR}" -cu "${CUDA}"

            else
                python src/scripts/Tagged_document.py -m ${MODEL} -id "${INPUTDIR}"
            fi
        

        else
            echo Not arguments the script requires at least model and input file
        fi

    else
        echo invalid option, USE for use a model, TRAIN for train a new model
    fi

fi